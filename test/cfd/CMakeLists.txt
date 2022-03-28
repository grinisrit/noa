####################################################################################################
##################                      CFD functional tests                     ###################
####################################################################################################

# Gflags
include(fetch-gflags)

# Find OpenMP
find_package(OpenMP QUIET)

# Find ZLIB
find_package(ZLIB)


# Main executables
add_executable(mass_lumping main.cc)
set_target_properties(muon_model PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link libraries
target_include_directories(mass_lumping PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(mass_lumping PRIVATE ${PROJECT_NAME} gflags ${OpenMP_CXX_LIBRARIES} ZLIB::ZLIB)
target_compile_options(mass_lumping PRIVATE
        PRIVATE -O3
        $<$<COMPILE_LANGUAGE:CXX>:${W_FLAGS} ${OpenMP_CXX_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${MCXX_CUDA}>)