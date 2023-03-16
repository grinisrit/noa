# remove existing files under the install prefix
install( CODE "MESSAGE( \"-- Removing existing include directory: ${CMAKE_INSTALL_FULL_INCLUDEDIR}/TNL/\")" )
install( CODE "file( REMOVE_RECURSE \"${CMAKE_INSTALL_FULL_INCLUDEDIR}/TNL/\" )" )

# install the header files
install( CODE "MESSAGE( \"-- Installing include directory: ${CMAKE_INSTALL_FULL_INCLUDEDIR}/TNL/\")" )
install( DIRECTORY TNL TYPE INCLUDE
         MESSAGE_NEVER
         FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" )

add_subdirectory( Python )

if( TNL_BUILD_BENCHMARKS )
   add_subdirectory( Benchmarks )
endif()

if( TNL_BUILD_EXAMPLES )
   add_subdirectory( Examples )
endif()

if( TNL_BUILD_TOOLS )
   add_subdirectory( Tools )
endif()

if( TNL_BUILD_TESTS )
   add_subdirectory( UnitTests )
endif()
if( TNL_BUILD_MATRIX_TESTS )
   add_subdirectory( UnitTests/Matrices )
endif()
