# enable C++14 for pytnl (due to py::overload_cast)
set(PYBIND11_CPP_STANDARD -std=c++14)

set( sources
      Grid1D.cpp
      Grid2D.cpp
      Grid3D.cpp
      Mesh.cpp
      MeshReaders.cpp
      MeshWriters.cpp
      Object.cpp
      SparseMatrix.cpp
      String.cpp
      VTKTraits.cpp
      tnl.cpp
)
pybind11_add_module( pytnl ${sources} )

# rename the shared library to tnl.cpython-XXm-x86_64-linux-gnu.so
set_target_properties( pytnl PROPERTIES LIBRARY_OUTPUT_NAME tnl DEBUG_POSTFIX "_dbg" )

# indicate the postfix to the target so that the pybind11 module name can be set accordingly
if( CMAKE_BUILD_TYPE STREQUAL "Debug")
   target_compile_options( pytnl PRIVATE -DPYTNL_MODULE_POSTFIX=_dbg )
endif()

# Skip -march=native -mtune=native for pytnl - optimizing python bindings for
# a specific architecture is not very useful and prevents using Python tools on
# hybrid clusters.
get_target_property( pytnl_COMPILE_OPTIONS pytnl COMPILE_OPTIONS )
if( pytnl_COMPILE_OPTIONS )
   string( REPLACE "-march=native" "" pytnl_COMPILE_OPTIONS "${pytnl_COMPILE_OPTIONS}" )
   string( REPLACE "-mtune=native" "" pytnl_COMPILE_OPTIONS "${pytnl_COMPILE_OPTIONS}" )
   set_target_properties( pytnl PROPERTIES COMPILE_OPTIONS "${pytnl_COMPILE_OPTIONS}" )
endif()

# We have bindings for unsafe objects (e.g. Array::operator[]) where assertion
# is the only safeguard, so we need to translate the TNL::AssertionError to
# Python's AssertionError.
# NDEBUG is defined in the global CMAKE_CXX_FLAGS and cannot be easily removed
# per-target, so we need to undefine it by passing -U NDEBUG.
target_compile_options( pytnl PRIVATE -U NDEBUG -D TNL_THROW_ASSERTION_ERROR )

# disable errors due to -Wunused-value coming from pybind11
if( ${WITH_CI_FLAGS} )
   if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      target_compile_options( pytnl PRIVATE -Wno-error=unused-value )
   endif()
endif()


# enable zlib and tinyxml2 (used by VTUReader)
find_package( ZLIB )
if( ZLIB_FOUND )
   target_compile_definitions(pytnl PUBLIC "-DHAVE_ZLIB")
   target_include_directories(pytnl PUBLIC ${ZLIB_INCLUDE_DIRS})
   target_link_libraries(pytnl PUBLIC ${ZLIB_LIBRARIES})
endif()

find_package( tinyxml2 QUIET )
if( tinyxml2_FOUND )
   target_compile_definitions(pytnl PUBLIC "-DHAVE_TINYXML2")
   target_link_libraries(pytnl PUBLIC tinyxml2::tinyxml2)
endif()


install( TARGETS pytnl DESTINATION ${PYTHON_SITE_PACKAGES_DIR} )
