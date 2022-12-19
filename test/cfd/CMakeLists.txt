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


# Link libraries
target_include_directories(mass_lumping PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(mass_lumping PRIVATE ${PROJECT_NAME} gflags ${OpenMP_CXX_LIBRARIES} ZLIB::ZLIB)
target_precompile_headers(mass_lumping PRIVATE main-pch.hh)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	message("[mass_lumping] Debug mode, OpenMP is off")
	target_compile_options(mass_lumping PRIVATE -DHAVE_ZLIB)
else()
	target_compile_options(mass_lumping
		PRIVATE -DHAVE_ZLIB -O3 -DHAVE_OPENMP $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
	)
endif()
