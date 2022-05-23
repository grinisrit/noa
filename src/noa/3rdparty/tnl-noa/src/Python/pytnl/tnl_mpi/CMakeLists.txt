# enable C++14 for pytnl_mpi (due to py::overload_cast)
set(PYBIND11_CPP_STANDARD -std=c++14)

set( sources
      DistributedMesh.cpp
      DistributedMeshReaders.cpp
      DistributedMeshWriters.cpp
      tnl_mpi.cpp
)
pybind11_add_module( pytnl_mpi ${sources} )

# rename the shared library to tnl_mpi.cpython-XXm-x86_64-linux-gnu.so
set_target_properties( pytnl_mpi PROPERTIES LIBRARY_OUTPUT_NAME tnl_mpi DEBUG_POSTFIX "_dbg" )

# indicate the postfix to the target so that the pybind11 module name can be set accordingly
if( CMAKE_BUILD_TYPE STREQUAL "Debug")
   target_compile_options( pytnl_mpi PRIVATE -DPYTNL_MODULE_POSTFIX=_dbg )
endif()

# Skip -march=native -mtune=native for pytnl_mpi - optimizing python bindings for
# a specific architecture is not very useful and prevents using Python tools on
# hybrid clusters.
get_target_property( pytnl_mpi_COMPILE_OPTIONS pytnl_mpi COMPILE_OPTIONS )
if( pytnl_mpi_COMPILE_OPTIONS )
   string( REPLACE "-march=native" "" pytnl_mpi_COMPILE_OPTIONS "${pytnl_mpi_COMPILE_OPTIONS}" )
   string( REPLACE "-mtune=native" "" pytnl_mpi_COMPILE_OPTIONS "${pytnl_mpi_COMPILE_OPTIONS}" )
   set_target_properties( pytnl_mpi PROPERTIES COMPILE_OPTIONS "${pytnl_mpi_COMPILE_OPTIONS}" )
endif()

# We have bindings for unsafe objects (e.g. Array::operator[]) where assertion
# is the only safeguard, so we need to translate the TNL::AssertionError to
# Python's AssertionError.
# NDEBUG is defined in the global CMAKE_CXX_FLAGS and cannot be easily removed
# per-target, so we need to undefine it by passing -U NDEBUG.
target_compile_options( pytnl_mpi PRIVATE -U NDEBUG -D TNL_THROW_ASSERTION_ERROR )

# disable errors due to -Wunused-value coming from pybind11
if( ${WITH_CI_FLAGS} )
   if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      target_compile_options( pytnl_mpi PRIVATE -Wno-error=unused-value )
   endif()
endif()


# enable zlib and tinyxml2 (used by PVTUReader)
find_package( ZLIB )
if( ZLIB_FOUND )
   target_compile_definitions(pytnl_mpi PUBLIC "-DHAVE_ZLIB")
   target_include_directories(pytnl_mpi PUBLIC ${ZLIB_INCLUDE_DIRS})
   target_link_libraries(pytnl_mpi PUBLIC ${ZLIB_LIBRARIES})
endif()

find_package( tinyxml2 QUIET )
if( tinyxml2_FOUND )
   target_compile_definitions(pytnl_mpi PUBLIC "-DHAVE_TINYXML2")
   target_link_libraries(pytnl_mpi PUBLIC tinyxml2::tinyxml2)
endif()


install( TARGETS pytnl_mpi DESTINATION ${PYTHON_SITE_PACKAGES_DIR} )
