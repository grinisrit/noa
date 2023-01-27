####################################################################################################
##################                      PMS functional tests                     ###################
####################################################################################################

# Gflags
include(fetch-gflags)

# Find ZLIB
find_package(ZLIB REQUIRED)

# Fetch materials
include(pumas-materials)

# Main executables
add_executable(muon_model
        muon-model.cc)
        
add_dependencies(muon_model pumas_materials)

# Link libraries
target_link_libraries(muon_model PRIVATE ${PROJECT_NAME} gflags ZLIB::ZLIB)
target_compile_options(muon_model PRIVATE
        PRIVATE -O3 -DHAVE_ZLIB
	$<$<COMPILE_LANGUAGE:CXX>:${W_FLAGS} -fpermissive>)
target_add_openmp( muon_model )
