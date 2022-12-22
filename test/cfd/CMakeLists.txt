####################################################################################################
##################                      CFD functional tests                     ###################
####################################################################################################

# Gflags
include(fetch-gflags)

# Find ZLIB
find_package(ZLIB)


# Main executables
add_executable(mass_lumping main.cc)


# Link libraries
target_link_libraries(mass_lumping PRIVATE ${PROJECT_NAME} gflags ZLIB::ZLIB)
target_precompile_headers(mass_lumping PRIVATE main-pch.hh)
target_compile_options( mass_lumping PRIVATE -DHAVE_ZLIB -O3 )
target_add_openmp( mass_lumping )
